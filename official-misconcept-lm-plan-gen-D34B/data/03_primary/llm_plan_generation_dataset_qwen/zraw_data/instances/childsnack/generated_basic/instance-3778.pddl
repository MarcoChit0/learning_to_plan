; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 763853

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child432 - child
    bread338 - bread-portion
    content429 - content-portion
    tray462 - tray
    table318 table422 table96 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray462 kitchen)
     (at_kitchen_bread bread338)
     (at_kitchen_content content429)
     (not_allergic_gluten child432)
     (waiting child432 table318)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child432)
    )
  )
)
