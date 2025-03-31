; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 942583

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child484 child273 - child
    bread14 bread131 - bread-portion
    content188 content440 - content-portion
    tray236 tray163 tray392 - tray
    table226 table273 table298 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray236 kitchen)
     (at tray163 kitchen)
     (at tray392 kitchen)
     (at_kitchen_bread bread14)
     (at_kitchen_bread bread131)
     (at_kitchen_content content188)
     (at_kitchen_content content440)
     (no_gluten_bread bread14)
     (no_gluten_content content440)
     (allergic_gluten child484)
     (not_allergic_gluten child273)
     (waiting child484 table273)
     (waiting child273 table298)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child484)
     (served child273)
    )
  )
)
