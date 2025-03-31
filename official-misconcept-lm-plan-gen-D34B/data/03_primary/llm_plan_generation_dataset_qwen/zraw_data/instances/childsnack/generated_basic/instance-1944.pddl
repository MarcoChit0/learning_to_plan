; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 906348

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child143 - child
    bread185 - bread-portion
    content267 - content-portion
    tray426 tray435 tray220 - tray
    table183 table378 table196 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray426 kitchen)
     (at tray435 kitchen)
     (at tray220 kitchen)
     (at_kitchen_bread bread185)
     (at_kitchen_content content267)
     (not_allergic_gluten child143)
     (waiting child143 table196)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child143)
    )
  )
)
