; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 935282

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child407 - child
    bread134 - bread-portion
    content425 - content-portion
    tray496 - tray
    table353 table164 table359 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray496 kitchen)
     (at_kitchen_bread bread134)
     (at_kitchen_content content425)
     (not_allergic_gluten child407)
     (waiting child407 table164)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child407)
    )
  )
)
