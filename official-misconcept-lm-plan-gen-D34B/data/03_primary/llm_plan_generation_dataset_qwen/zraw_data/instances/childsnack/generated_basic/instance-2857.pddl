; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 288082

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child203 - child
    bread485 - bread-portion
    content263 - content-portion
    tray483 tray259 - tray
    table214 table94 table159 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray483 kitchen)
     (at tray259 kitchen)
     (at_kitchen_bread bread485)
     (at_kitchen_content content263)
     (not_allergic_gluten child203)
     (waiting child203 table214)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child203)
    )
  )
)
