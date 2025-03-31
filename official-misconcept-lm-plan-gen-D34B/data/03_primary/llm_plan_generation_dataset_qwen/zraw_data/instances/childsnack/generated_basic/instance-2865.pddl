; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 723444

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child277 - child
    bread72 - bread-portion
    content13 - content-portion
    tray21 tray19 - tray
    table364 table281 table331 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray21 kitchen)
     (at tray19 kitchen)
     (at_kitchen_bread bread72)
     (at_kitchen_content content13)
     (not_allergic_gluten child277)
     (waiting child277 table364)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child277)
    )
  )
)
