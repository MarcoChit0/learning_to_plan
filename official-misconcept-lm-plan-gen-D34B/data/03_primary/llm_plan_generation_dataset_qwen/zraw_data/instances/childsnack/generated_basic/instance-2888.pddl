; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 226215

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child53 - child
    bread296 - bread-portion
    content104 - content-portion
    tray266 tray380 - tray
    table151 table8 table193 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray266 kitchen)
     (at tray380 kitchen)
     (at_kitchen_bread bread296)
     (at_kitchen_content content104)
     (not_allergic_gluten child53)
     (waiting child53 table8)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child53)
    )
  )
)
