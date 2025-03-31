; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 787980

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child36 - child
    bread121 - bread-portion
    content75 - content-portion
    tray133 tray381 tray146 tray14 - tray
    table222 table168 table421 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray133 kitchen)
     (at tray381 kitchen)
     (at tray146 kitchen)
     (at tray14 kitchen)
     (at_kitchen_bread bread121)
     (at_kitchen_content content75)
     (not_allergic_gluten child36)
     (waiting child36 table222)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child36)
    )
  )
)
