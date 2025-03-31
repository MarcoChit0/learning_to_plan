; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 348992

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child28 child316 - child
    bread329 bread358 - bread-portion
    content63 content182 - content-portion
    tray383 tray201 - tray
    table206 table232 table8 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray383 kitchen)
     (at tray201 kitchen)
     (at_kitchen_bread bread329)
     (at_kitchen_bread bread358)
     (at_kitchen_content content63)
     (at_kitchen_content content182)
     (not_allergic_gluten child28)
     (not_allergic_gluten child316)
     (waiting child28 table8)
     (waiting child316 table8)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child28)
     (served child316)
    )
  )
)
