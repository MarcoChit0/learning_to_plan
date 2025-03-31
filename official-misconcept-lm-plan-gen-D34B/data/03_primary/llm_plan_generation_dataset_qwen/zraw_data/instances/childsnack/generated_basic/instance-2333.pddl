; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 228693

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child327 - child
    bread250 - bread-portion
    content311 - content-portion
    tray127 - tray
    table140 table389 table33 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray127 kitchen)
     (at_kitchen_bread bread250)
     (at_kitchen_content content311)
     (not_allergic_gluten child327)
     (waiting child327 table389)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child327)
    )
  )
)
