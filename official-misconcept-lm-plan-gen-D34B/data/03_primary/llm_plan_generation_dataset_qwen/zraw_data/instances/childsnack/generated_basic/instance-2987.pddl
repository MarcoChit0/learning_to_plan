; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 710602

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child357 child340 - child
    bread32 bread389 - bread-portion
    content170 content410 - content-portion
    tray91 tray63 tray241 tray476 - tray
    table92 table227 table194 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray91 kitchen)
     (at tray63 kitchen)
     (at tray241 kitchen)
     (at tray476 kitchen)
     (at_kitchen_bread bread32)
     (at_kitchen_bread bread389)
     (at_kitchen_content content170)
     (at_kitchen_content content410)
     (not_allergic_gluten child357)
     (not_allergic_gluten child340)
     (waiting child357 table194)
     (waiting child340 table194)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child357)
     (served child340)
    )
  )
)
