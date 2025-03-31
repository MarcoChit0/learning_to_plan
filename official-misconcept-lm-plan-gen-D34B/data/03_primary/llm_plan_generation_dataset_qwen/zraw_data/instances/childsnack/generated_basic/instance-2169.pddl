; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 733567

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child67 - child
    bread60 - bread-portion
    content215 - content-portion
    tray279 - tray
    table462 table432 table494 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray279 kitchen)
     (at_kitchen_bread bread60)
     (at_kitchen_content content215)
     (not_allergic_gluten child67)
     (waiting child67 table462)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child67)
    )
  )
)
