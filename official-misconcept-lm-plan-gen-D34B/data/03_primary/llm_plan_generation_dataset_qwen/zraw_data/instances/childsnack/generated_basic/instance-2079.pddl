; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 253724

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child271 - child
    bread82 - bread-portion
    content161 - content-portion
    tray146 tray428 tray495 tray282 - tray
    table227 table463 table326 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray146 kitchen)
     (at tray428 kitchen)
     (at tray495 kitchen)
     (at tray282 kitchen)
     (at_kitchen_bread bread82)
     (at_kitchen_content content161)
     (not_allergic_gluten child271)
     (waiting child271 table463)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child271)
    )
  )
)
