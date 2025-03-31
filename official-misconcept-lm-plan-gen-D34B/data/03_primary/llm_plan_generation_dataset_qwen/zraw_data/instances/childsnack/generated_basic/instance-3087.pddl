; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 835906

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child92 child39 - child
    bread135 bread51 - bread-portion
    content288 content31 - content-portion
    tray180 tray58 tray251 - tray
    table69 table348 table222 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray180 kitchen)
     (at tray58 kitchen)
     (at tray251 kitchen)
     (at_kitchen_bread bread135)
     (at_kitchen_bread bread51)
     (at_kitchen_content content288)
     (at_kitchen_content content31)
     (not_allergic_gluten child92)
     (not_allergic_gluten child39)
     (waiting child92 table222)
     (waiting child39 table348)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child92)
     (served child39)
    )
  )
)
