; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 152642

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child485 - child
    bread29 - bread-portion
    content267 - content-portion
    tray485 tray7 tray190 tray249 - tray
    table310 table375 table44 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray485 kitchen)
     (at tray7 kitchen)
     (at tray190 kitchen)
     (at tray249 kitchen)
     (at_kitchen_bread bread29)
     (at_kitchen_content content267)
     (not_allergic_gluten child485)
     (waiting child485 table375)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child485)
    )
  )
)
