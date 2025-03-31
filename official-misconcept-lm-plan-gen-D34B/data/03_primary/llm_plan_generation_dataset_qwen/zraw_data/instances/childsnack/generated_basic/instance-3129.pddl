; child-snack task with 2 children and 0.65 gluten factor 
; constant factor of 1.3
; random seed: 626694

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child160 child296 - child
    bread324 bread415 - bread-portion
    content153 content290 - content-portion
    tray440 tray347 tray453 tray432 - tray
    table308 table19 table342 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray440 kitchen)
     (at tray347 kitchen)
     (at tray453 kitchen)
     (at tray432 kitchen)
     (at_kitchen_bread bread324)
     (at_kitchen_bread bread415)
     (at_kitchen_content content153)
     (at_kitchen_content content290)
     (no_gluten_bread bread415)
     (no_gluten_content content290)
     (allergic_gluten child296)
     (not_allergic_gluten child160)
     (waiting child160 table19)
     (waiting child296 table342)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child160)
     (served child296)
    )
  )
)
