; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 758910

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child217 child232 - child
    bread334 bread120 - bread-portion
    content88 content131 - content-portion
    tray146 tray64 tray185 - tray
    table376 table202 table206 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray146 kitchen)
     (at tray64 kitchen)
     (at tray185 kitchen)
     (at_kitchen_bread bread334)
     (at_kitchen_bread bread120)
     (at_kitchen_content content88)
     (at_kitchen_content content131)
     (not_allergic_gluten child232)
     (not_allergic_gluten child217)
     (waiting child217 table202)
     (waiting child232 table376)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child217)
     (served child232)
    )
  )
)
