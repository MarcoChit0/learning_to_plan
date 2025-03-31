; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 21177

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child174 - child
    bread201 - bread-portion
    content496 - content-portion
    tray225 tray473 tray255 - tray
    table26 table285 table326 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray225 kitchen)
     (at tray473 kitchen)
     (at tray255 kitchen)
     (at_kitchen_bread bread201)
     (at_kitchen_content content496)
     (not_allergic_gluten child174)
     (waiting child174 table326)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child174)
    )
  )
)
