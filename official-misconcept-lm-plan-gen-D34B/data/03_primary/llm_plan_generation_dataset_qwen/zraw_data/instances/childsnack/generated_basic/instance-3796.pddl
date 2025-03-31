; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 160245

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child446 - child
    bread177 - bread-portion
    content78 - content-portion
    tray478 - tray
    table323 table124 table15 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray478 kitchen)
     (at_kitchen_bread bread177)
     (at_kitchen_content content78)
     (not_allergic_gluten child446)
     (waiting child446 table323)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child446)
    )
  )
)
