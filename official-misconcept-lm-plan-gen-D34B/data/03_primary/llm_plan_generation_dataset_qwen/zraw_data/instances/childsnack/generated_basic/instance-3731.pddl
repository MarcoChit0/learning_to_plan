; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 151332

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child126 - child
    bread231 - bread-portion
    content297 - content-portion
    tray314 tray282 tray187 - tray
    table422 table92 table392 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray314 kitchen)
     (at tray282 kitchen)
     (at tray187 kitchen)
     (at_kitchen_bread bread231)
     (at_kitchen_content content297)
     (not_allergic_gluten child126)
     (waiting child126 table422)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child126)
    )
  )
)
