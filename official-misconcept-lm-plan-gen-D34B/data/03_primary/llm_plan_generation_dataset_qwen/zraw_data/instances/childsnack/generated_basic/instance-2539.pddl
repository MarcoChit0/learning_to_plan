; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 731601

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child163 - child
    bread459 - bread-portion
    content363 - content-portion
    tray358 tray465 tray58 - tray
    table148 table395 table486 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray358 kitchen)
     (at tray465 kitchen)
     (at tray58 kitchen)
     (at_kitchen_bread bread459)
     (at_kitchen_content content363)
     (not_allergic_gluten child163)
     (waiting child163 table395)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child163)
    )
  )
)
