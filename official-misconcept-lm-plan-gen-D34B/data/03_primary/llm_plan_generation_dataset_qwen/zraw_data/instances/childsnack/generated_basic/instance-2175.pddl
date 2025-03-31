; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 898753

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child409 - child
    bread170 - bread-portion
    content60 - content-portion
    tray19 - tray
    table202 table432 table311 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray19 kitchen)
     (at_kitchen_bread bread170)
     (at_kitchen_content content60)
     (not_allergic_gluten child409)
     (waiting child409 table311)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child409)
    )
  )
)
