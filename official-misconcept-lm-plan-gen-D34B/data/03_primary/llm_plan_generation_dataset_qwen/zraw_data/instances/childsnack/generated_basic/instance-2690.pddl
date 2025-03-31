; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 840960

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child84 child284 child443 - child
    bread381 bread138 bread463 - bread-portion
    content445 content48 content91 - content-portion
    tray449 tray198 tray213 - tray
    table19 table55 table41 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray449 kitchen)
     (at tray198 kitchen)
     (at tray213 kitchen)
     (at_kitchen_bread bread381)
     (at_kitchen_bread bread138)
     (at_kitchen_bread bread463)
     (at_kitchen_content content445)
     (at_kitchen_content content48)
     (at_kitchen_content content91)
     (not_allergic_gluten child443)
     (not_allergic_gluten child284)
     (not_allergic_gluten child84)
     (waiting child84 table41)
     (waiting child284 table19)
     (waiting child443 table55)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child84)
     (served child284)
     (served child443)
    )
  )
)
