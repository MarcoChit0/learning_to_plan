; child-snack task with 2 children and 0.71 gluten factor 
; constant factor of 1.7
; random seed: 402971

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child157 child382 - child
    bread41 bread473 - bread-portion
    content317 content155 - content-portion
    tray133 tray378 tray289 tray265 - tray
    table206 table191 table36 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray133 kitchen)
     (at tray378 kitchen)
     (at tray289 kitchen)
     (at tray265 kitchen)
     (at_kitchen_bread bread41)
     (at_kitchen_bread bread473)
     (at_kitchen_content content317)
     (at_kitchen_content content155)
     (no_gluten_bread bread41)
     (no_gluten_content content317)
     (allergic_gluten child157)
     (not_allergic_gluten child382)
     (waiting child157 table191)
     (waiting child382 table206)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child157)
     (served child382)
    )
  )
)
