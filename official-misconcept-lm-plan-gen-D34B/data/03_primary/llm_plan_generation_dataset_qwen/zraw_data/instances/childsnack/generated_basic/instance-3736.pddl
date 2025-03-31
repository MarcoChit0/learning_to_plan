; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 844466

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child278 - child
    bread277 - bread-portion
    content323 - content-portion
    tray140 tray444 tray43 - tray
    table300 table199 table416 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray140 kitchen)
     (at tray444 kitchen)
     (at tray43 kitchen)
     (at_kitchen_bread bread277)
     (at_kitchen_content content323)
     (not_allergic_gluten child278)
     (waiting child278 table416)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child278)
    )
  )
)
