; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 132374

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child122 - child
    bread322 - bread-portion
    content338 - content-portion
    tray253 - tray
    table76 table262 table248 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray253 kitchen)
     (at_kitchen_bread bread322)
     (at_kitchen_content content338)
     (not_allergic_gluten child122)
     (waiting child122 table248)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child122)
    )
  )
)
