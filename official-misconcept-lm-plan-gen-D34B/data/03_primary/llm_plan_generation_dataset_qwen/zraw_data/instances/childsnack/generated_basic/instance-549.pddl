; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 308948

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child274 child412 - child
    bread18 bread344 - bread-portion
    content355 content322 - content-portion
    tray79 - tray
    table119 table12 table419 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray79 kitchen)
     (at_kitchen_bread bread18)
     (at_kitchen_bread bread344)
     (at_kitchen_content content355)
     (at_kitchen_content content322)
     (no_gluten_bread bread18)
     (no_gluten_content content322)
     (allergic_gluten child274)
     (not_allergic_gluten child412)
     (waiting child274 table419)
     (waiting child412 table419)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child274)
     (served child412)
    )
  )
)
