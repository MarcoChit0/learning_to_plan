; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 85697

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child171 child56 - child
    bread241 bread296 - bread-portion
    content92 content347 - content-portion
    tray435 tray173 tray187 - tray
    table365 table417 table392 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray435 kitchen)
     (at tray173 kitchen)
     (at tray187 kitchen)
     (at_kitchen_bread bread241)
     (at_kitchen_bread bread296)
     (at_kitchen_content content92)
     (at_kitchen_content content347)
     (not_allergic_gluten child171)
     (not_allergic_gluten child56)
     (waiting child171 table365)
     (waiting child56 table392)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child171)
     (served child56)
    )
  )
)
