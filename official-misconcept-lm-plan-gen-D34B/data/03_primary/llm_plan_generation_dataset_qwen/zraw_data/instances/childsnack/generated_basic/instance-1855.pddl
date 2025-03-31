; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 310825

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child448 child71 - child
    bread236 bread80 - bread-portion
    content178 content464 - content-portion
    tray394 - tray
    table398 table78 table374 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray394 kitchen)
     (at_kitchen_bread bread236)
     (at_kitchen_bread bread80)
     (at_kitchen_content content178)
     (at_kitchen_content content464)
     (no_gluten_bread bread80)
     (no_gluten_content content178)
     (allergic_gluten child448)
     (not_allergic_gluten child71)
     (waiting child448 table398)
     (waiting child71 table398)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child448)
     (served child71)
    )
  )
)
