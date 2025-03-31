; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 901347

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child23 - child
    bread476 - bread-portion
    content316 - content-portion
    tray77 tray474 tray338 - tray
    table464 table469 table90 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray77 kitchen)
     (at tray474 kitchen)
     (at tray338 kitchen)
     (at_kitchen_bread bread476)
     (at_kitchen_content content316)
     (not_allergic_gluten child23)
     (waiting child23 table464)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child23)
    )
  )
)
