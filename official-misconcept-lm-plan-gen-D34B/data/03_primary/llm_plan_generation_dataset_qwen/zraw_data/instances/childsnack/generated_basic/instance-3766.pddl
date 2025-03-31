; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 112785

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child440 - child
    bread2 - bread-portion
    content154 - content-portion
    tray409 - tray
    table251 table315 table417 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray409 kitchen)
     (at_kitchen_bread bread2)
     (at_kitchen_content content154)
     (not_allergic_gluten child440)
     (waiting child440 table315)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child440)
    )
  )
)
