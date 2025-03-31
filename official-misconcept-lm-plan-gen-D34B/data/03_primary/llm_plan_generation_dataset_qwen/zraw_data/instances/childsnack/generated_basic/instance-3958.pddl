; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 567299

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child188 child75 - child
    bread25 bread172 - bread-portion
    content189 content363 - content-portion
    tray65 - tray
    table66 table334 table84 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray65 kitchen)
     (at_kitchen_bread bread25)
     (at_kitchen_bread bread172)
     (at_kitchen_content content189)
     (at_kitchen_content content363)
     (no_gluten_bread bread172)
     (no_gluten_content content363)
     (allergic_gluten child188)
     (not_allergic_gluten child75)
     (waiting child188 table84)
     (waiting child75 table334)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child188)
     (served child75)
    )
  )
)
