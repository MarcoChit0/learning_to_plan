; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 457813

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child65 - child
    bread121 - bread-portion
    content438 - content-portion
    tray463 tray423 tray290 - tray
    table282 table349 table153 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray463 kitchen)
     (at tray423 kitchen)
     (at tray290 kitchen)
     (at_kitchen_bread bread121)
     (at_kitchen_content content438)
     (not_allergic_gluten child65)
     (waiting child65 table282)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child65)
    )
  )
)
