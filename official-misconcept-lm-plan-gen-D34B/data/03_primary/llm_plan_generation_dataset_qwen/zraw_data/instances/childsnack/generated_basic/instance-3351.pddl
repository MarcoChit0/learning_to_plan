; child-snack task with 2 children and 0.6 gluten factor 
; constant factor of 1.3
; random seed: 403565

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child475 child4 - child
    bread36 bread76 - bread-portion
    content114 content264 - content-portion
    tray57 tray289 - tray
    table252 table108 table365 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray57 kitchen)
     (at tray289 kitchen)
     (at_kitchen_bread bread36)
     (at_kitchen_bread bread76)
     (at_kitchen_content content114)
     (at_kitchen_content content264)
     (no_gluten_bread bread36)
     (no_gluten_content content114)
     (allergic_gluten child475)
     (not_allergic_gluten child4)
     (waiting child475 table252)
     (waiting child4 table108)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child475)
     (served child4)
    )
  )
)
