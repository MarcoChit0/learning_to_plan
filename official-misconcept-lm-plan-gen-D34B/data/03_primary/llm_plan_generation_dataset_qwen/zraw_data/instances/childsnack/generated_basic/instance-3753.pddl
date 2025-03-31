; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 961210

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child33 - child
    bread117 - bread-portion
    content285 - content-portion
    tray462 tray401 tray328 - tray
    table437 table184 table405 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray462 kitchen)
     (at tray401 kitchen)
     (at tray328 kitchen)
     (at_kitchen_bread bread117)
     (at_kitchen_content content285)
     (not_allergic_gluten child33)
     (waiting child33 table437)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child33)
    )
  )
)
