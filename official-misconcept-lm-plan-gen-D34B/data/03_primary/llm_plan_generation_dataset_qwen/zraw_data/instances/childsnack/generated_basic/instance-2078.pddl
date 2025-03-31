; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 906495

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child91 - child
    bread241 - bread-portion
    content347 - content-portion
    tray263 tray462 tray282 tray35 - tray
    table446 table149 table410 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray263 kitchen)
     (at tray462 kitchen)
     (at tray282 kitchen)
     (at tray35 kitchen)
     (at_kitchen_bread bread241)
     (at_kitchen_content content347)
     (not_allergic_gluten child91)
     (waiting child91 table149)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child91)
    )
  )
)
