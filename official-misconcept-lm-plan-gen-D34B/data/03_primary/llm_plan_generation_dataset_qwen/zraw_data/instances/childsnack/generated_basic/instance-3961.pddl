; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 748080

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child42 - child
    bread483 - bread-portion
    content462 - content-portion
    tray473 tray233 tray478 - tray
    table25 table376 table258 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray473 kitchen)
     (at tray233 kitchen)
     (at tray478 kitchen)
     (at_kitchen_bread bread483)
     (at_kitchen_content content462)
     (not_allergic_gluten child42)
     (waiting child42 table376)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child42)
    )
  )
)
