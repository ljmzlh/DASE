\echo Use "ALTER EXTENSION vector UPDATE TO '0.8.3'" to load this file. \quit

DO $$
DECLARE
    fam_oid     oid;
    fam_schema  text;
BEGIN
    SELECT opc.opcfamily, n.nspname
      INTO fam_oid, fam_schema
      FROM pg_catalog.pg_opclass opc
      JOIN pg_catalog.pg_namespace n ON n.oid = opc.opcnamespace
     WHERE opc.opcname = 'vector_l2_ops'
       AND opc.opcmethod = (SELECT oid FROM pg_catalog.pg_am WHERE amname = 'hnsw')
     LIMIT 1;

    IF fam_oid IS NULL THEN
        RAISE EXCEPTION 'hnsw operator class % not found', 'vector_l2_ops';
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_amop
         WHERE amopfamily = fam_oid
           AND amopstrategy = 51
           AND amoplefttype = 'vector'::regtype
           AND amoprighttype = 'vector'::regtype
           AND amoppurpose = 'o'
    ) THEN
        UPDATE pg_catalog.pg_amop
           SET amoppurpose = 's'
         WHERE amopfamily = fam_oid
           AND amopstrategy = 51
           AND amoplefttype = 'vector'::regtype
           AND amoprighttype = 'vector'::regtype
           AND amoppurpose = 'o';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_amop
         WHERE amopfamily = fam_oid
           AND amopstrategy = 51
           AND amoplefttype = 'vector'::regtype
           AND amoprighttype = 'vector'::regtype
           AND amoppurpose = 's'
    ) THEN
        EXECUTE format(
            'ALTER OPERATOR FAMILY %1$I.vector_l2_ops USING hnsw ADD OPERATOR 51 %1$I.<->@ (vector, vector)',
            fam_schema);
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_amop
         WHERE amopfamily = fam_oid
           AND amopstrategy = 52
           AND amoplefttype = 'vector'::regtype
           AND amoprighttype = 'vector'::regtype
           AND amoppurpose = 'o'
    ) THEN
        UPDATE pg_catalog.pg_amop
           SET amoppurpose = 's'
         WHERE amopfamily = fam_oid
           AND amopstrategy = 52
           AND amoplefttype = 'vector'::regtype
           AND amoprighttype = 'vector'::regtype
           AND amoppurpose = 'o';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_amop
         WHERE amopfamily = fam_oid
           AND amopstrategy = 52
           AND amoplefttype = 'vector'::regtype
           AND amoprighttype = 'vector'::regtype
           AND amoppurpose = 's'
    ) THEN
        EXECUTE format(
            'ALTER OPERATOR FAMILY %1$I.vector_l2_ops USING hnsw ADD OPERATOR 52 %1$I.<-># (vector, vector)',
            fam_schema);
    END IF;
END
$$ LANGUAGE plpgsql;

