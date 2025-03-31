; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 267624

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child141 - child
    bread151 - bread-portion
    content244 - content-portion
    tray417 tray336 tray345 tray386 - tray
    table321 table434 table261 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray417 kitchen)
     (at tray336 kitchen)
     (at tray345 kitchen)
     (at tray386 kitchen)
     (at_kitchen_bread bread151)
     (at_kitchen_content content244)
     (not_allergic_gluten child141)
     (waiting child141 table434)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child141)
    )
  )
)
