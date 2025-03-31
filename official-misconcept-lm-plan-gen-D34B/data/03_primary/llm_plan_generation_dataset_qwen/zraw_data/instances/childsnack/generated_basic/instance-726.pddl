; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 787587

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child440 - child
    bread435 - bread-portion
    content285 - content-portion
    tray379 tray373 tray252 tray6 - tray
    table315 table11 table396 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray379 kitchen)
     (at tray373 kitchen)
     (at tray252 kitchen)
     (at tray6 kitchen)
     (at_kitchen_bread bread435)
     (at_kitchen_content content285)
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
