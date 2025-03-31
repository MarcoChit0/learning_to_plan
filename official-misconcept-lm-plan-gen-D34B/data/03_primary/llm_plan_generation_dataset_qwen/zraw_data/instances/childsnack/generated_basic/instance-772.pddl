; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 199593

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child165 child201 - child
    bread187 bread55 - bread-portion
    content174 content32 - content-portion
    tray441 tray254 - tray
    table249 table268 table176 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray441 kitchen)
     (at tray254 kitchen)
     (at_kitchen_bread bread187)
     (at_kitchen_bread bread55)
     (at_kitchen_content content174)
     (at_kitchen_content content32)
     (no_gluten_bread bread55)
     (no_gluten_content content174)
     (allergic_gluten child165)
     (not_allergic_gluten child201)
     (waiting child165 table176)
     (waiting child201 table176)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child165)
     (served child201)
    )
  )
)
