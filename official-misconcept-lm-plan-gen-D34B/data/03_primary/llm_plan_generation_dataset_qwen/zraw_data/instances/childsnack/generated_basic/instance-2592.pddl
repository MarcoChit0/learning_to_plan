; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 648517

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child132 - child
    bread426 - bread-portion
    content214 - content-portion
    tray482 tray250 tray82 - tray
    table99 table436 table339 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray482 kitchen)
     (at tray250 kitchen)
     (at tray82 kitchen)
     (at_kitchen_bread bread426)
     (at_kitchen_content content214)
     (not_allergic_gluten child132)
     (waiting child132 table339)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child132)
    )
  )
)
