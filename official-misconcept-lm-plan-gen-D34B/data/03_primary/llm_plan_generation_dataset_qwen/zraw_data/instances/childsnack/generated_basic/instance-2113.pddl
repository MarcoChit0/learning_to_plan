; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 955017

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child141 child428 - child
    bread190 bread58 - bread-portion
    content219 content386 - content-portion
    tray364 tray324 tray54 - tray
    table8 table40 table253 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray364 kitchen)
     (at tray324 kitchen)
     (at tray54 kitchen)
     (at_kitchen_bread bread190)
     (at_kitchen_bread bread58)
     (at_kitchen_content content219)
     (at_kitchen_content content386)
     (no_gluten_bread bread58)
     (no_gluten_content content386)
     (allergic_gluten child141)
     (not_allergic_gluten child428)
     (waiting child141 table8)
     (waiting child428 table8)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child141)
     (served child428)
    )
  )
)
