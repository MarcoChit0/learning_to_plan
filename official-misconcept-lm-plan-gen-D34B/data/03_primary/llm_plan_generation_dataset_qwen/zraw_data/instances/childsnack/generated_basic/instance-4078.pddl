; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 68320

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child141 - child
    bread96 - bread-portion
    content334 - content-portion
    tray215 tray282 tray187 - tray
    table63 table59 table68 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray215 kitchen)
     (at tray282 kitchen)
     (at tray187 kitchen)
     (at_kitchen_bread bread96)
     (at_kitchen_content content334)
     (not_allergic_gluten child141)
     (waiting child141 table68)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child141)
    )
  )
)
