; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 534092

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child387 - child
    bread294 - bread-portion
    content78 - content-portion
    tray229 tray201 - tray
    table142 table25 table110 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray229 kitchen)
     (at tray201 kitchen)
     (at_kitchen_bread bread294)
     (at_kitchen_content content78)
     (not_allergic_gluten child387)
     (waiting child387 table110)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child387)
    )
  )
)
