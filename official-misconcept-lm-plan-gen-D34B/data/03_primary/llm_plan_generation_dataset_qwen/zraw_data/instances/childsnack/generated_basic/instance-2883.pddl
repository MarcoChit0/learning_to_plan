; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 445539

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child467 - child
    bread206 - bread-portion
    content427 - content-portion
    tray420 tray442 - tray
    table22 table214 table376 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray420 kitchen)
     (at tray442 kitchen)
     (at_kitchen_bread bread206)
     (at_kitchen_content content427)
     (not_allergic_gluten child467)
     (waiting child467 table376)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child467)
    )
  )
)
