; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 253371

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child188 child178 - child
    bread211 bread89 - bread-portion
    content219 content229 - content-portion
    tray197 tray353 tray187 tray340 - tray
    table426 table78 table256 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray197 kitchen)
     (at tray353 kitchen)
     (at tray187 kitchen)
     (at tray340 kitchen)
     (at_kitchen_bread bread211)
     (at_kitchen_bread bread89)
     (at_kitchen_content content219)
     (at_kitchen_content content229)
     (not_allergic_gluten child188)
     (not_allergic_gluten child178)
     (waiting child188 table256)
     (waiting child178 table426)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child188)
     (served child178)
    )
  )
)
