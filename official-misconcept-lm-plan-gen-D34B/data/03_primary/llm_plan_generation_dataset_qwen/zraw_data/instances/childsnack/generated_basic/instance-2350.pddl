; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 204856

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child30 - child
    bread65 - bread-portion
    content485 - content-portion
    tray122 - tray
    table151 table286 table491 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray122 kitchen)
     (at_kitchen_bread bread65)
     (at_kitchen_content content485)
     (not_allergic_gluten child30)
     (waiting child30 table286)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child30)
    )
  )
)
