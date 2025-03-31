; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 556456

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child266 - child
    bread259 - bread-portion
    content254 - content-portion
    tray237 - tray
    table325 table288 table206 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray237 kitchen)
     (at_kitchen_bread bread259)
     (at_kitchen_content content254)
     (not_allergic_gluten child266)
     (waiting child266 table206)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child266)
    )
  )
)
