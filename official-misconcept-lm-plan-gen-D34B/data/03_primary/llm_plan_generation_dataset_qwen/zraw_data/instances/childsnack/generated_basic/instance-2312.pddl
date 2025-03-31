; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 347302

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child438 - child
    bread353 - bread-portion
    content379 - content-portion
    tray446 - tray
    table193 table18 table428 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray446 kitchen)
     (at_kitchen_bread bread353)
     (at_kitchen_content content379)
     (not_allergic_gluten child438)
     (waiting child438 table18)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child438)
    )
  )
)
