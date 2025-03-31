; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 120249

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child272 child300 - child
    bread205 bread170 - bread-portion
    content481 content155 - content-portion
    tray18 tray325 tray429 - tray
    table58 table323 table212 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray18 kitchen)
     (at tray325 kitchen)
     (at tray429 kitchen)
     (at_kitchen_bread bread205)
     (at_kitchen_bread bread170)
     (at_kitchen_content content481)
     (at_kitchen_content content155)
     (not_allergic_gluten child272)
     (not_allergic_gluten child300)
     (waiting child272 table212)
     (waiting child300 table323)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child272)
     (served child300)
    )
  )
)
