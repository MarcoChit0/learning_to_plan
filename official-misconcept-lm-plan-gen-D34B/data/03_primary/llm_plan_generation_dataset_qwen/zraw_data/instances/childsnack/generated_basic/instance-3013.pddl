; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 579107

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child260 child157 - child
    bread223 bread220 - bread-portion
    content224 content259 - content-portion
    tray368 tray227 tray89 tray151 - tray
    table68 table231 table27 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray368 kitchen)
     (at tray227 kitchen)
     (at tray89 kitchen)
     (at tray151 kitchen)
     (at_kitchen_bread bread223)
     (at_kitchen_bread bread220)
     (at_kitchen_content content224)
     (at_kitchen_content content259)
     (not_allergic_gluten child157)
     (not_allergic_gluten child260)
     (waiting child260 table27)
     (waiting child157 table231)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child260)
     (served child157)
    )
  )
)
