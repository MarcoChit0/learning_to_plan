; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 407647

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child121 child346 - child
    bread353 bread390 - bread-portion
    content317 content127 - content-portion
    tray129 tray411 tray256 tray171 - tray
    table336 table170 table295 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray129 kitchen)
     (at tray411 kitchen)
     (at tray256 kitchen)
     (at tray171 kitchen)
     (at_kitchen_bread bread353)
     (at_kitchen_bread bread390)
     (at_kitchen_content content317)
     (at_kitchen_content content127)
     (no_gluten_bread bread353)
     (no_gluten_content content317)
     (allergic_gluten child121)
     (not_allergic_gluten child346)
     (waiting child121 table295)
     (waiting child346 table336)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child121)
     (served child346)
    )
  )
)
