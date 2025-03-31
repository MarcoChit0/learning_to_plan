; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 305254

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child389 - child
    bread67 - bread-portion
    content256 - content-portion
    tray117 tray477 tray496 tray67 - tray
    table227 table191 table494 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray117 kitchen)
     (at tray477 kitchen)
     (at tray496 kitchen)
     (at tray67 kitchen)
     (at_kitchen_bread bread67)
     (at_kitchen_content content256)
     (not_allergic_gluten child389)
     (waiting child389 table227)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child389)
    )
  )
)
