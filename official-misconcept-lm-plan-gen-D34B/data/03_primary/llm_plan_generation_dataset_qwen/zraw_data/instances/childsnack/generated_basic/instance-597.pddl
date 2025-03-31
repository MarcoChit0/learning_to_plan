; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 519599

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child226 - child
    bread191 - bread-portion
    content229 - content-portion
    tray483 - tray
    table79 table407 table30 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray483 kitchen)
     (at_kitchen_bread bread191)
     (at_kitchen_content content229)
     (not_allergic_gluten child226)
     (waiting child226 table79)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child226)
    )
  )
)
